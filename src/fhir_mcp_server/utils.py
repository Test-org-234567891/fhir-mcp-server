# Copyright (c) 2025, WSO2 LLC. (https://www.wso2.com/) All Rights Reserved.

# WSO2 LLC. licenses this file to you under the Apache License,
# Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.

import aiohttp
import logging

from fhir_mcp_server.oauth import ServerConfigs
from toon_format import encode as toon_encode

from typing import Any, Dict, List, Optional
from fhirpy import AsyncFHIRClient
from fhirpathpy import compile as fhircompile
from mcp.shared._httpx_utils import create_mcp_http_client

logger: logging.Logger = logging.getLogger(__name__)


def _build_field_tree(paths: List[str]) -> Dict[str, Any]:
    """
    Convert dot-notation paths into a nested dict that mirrors the shape to extract.
    ["address.city", "address.postalCode"] becomes {"address": {"city": None, "postalCode": None}}.
    None marks a leaf: copy the full value at that key without going deeper.
    """
    tree: Dict[str, Any] = {}
    for path in paths:
        parts = path.split(".")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = None  # None = leaf, copy the full value as-is
    return tree


def _filter_by_tree(obj: Any, tree: Dict[str, Any]) -> Any:
    """
    Walk obj against tree and return only the matching fields.
    When a value is a list (e.g. name, address in FHIR) the same tree
    is applied to each item in it, so dot-paths work the same either way.
    """
    if isinstance(obj, list):
        return [_filter_by_tree(i, tree) for i in obj if isinstance(i, dict)]
    if not isinstance(obj, dict):
        return obj
    out: Dict[str, Any] = {}
    for key, sub in tree.items():
        val = obj.get(key)
        if val is None:
            continue
        out[key] = val if sub is None else _filter_by_tree(val, sub)
    return out


def format_output(data: Any, fmt: str) -> Any:
    """Return TOON-encoded text for 'toon' format, or raw data for 'json'."""
    if fmt == "json":
        return data
    return toon_encode(data)


def filter_resource(
    resource: Dict[str, Any],
    fields: Dict[str, List[str]],
    mandatory_fields: List[str] = ["resourceType", "id"],
) -> Dict[str, Any]:
    """
    Return a filtered copy of resource, keeping only the paths in fields.
    Looks up paths by resourceType first, then merges in any '*' wildcard paths.
    mandatory_fields are always kept regardless of the paths list.
    Skips filtering entirely if no matching paths are found.
    """
    if not fields:
        return resource
    resource_type = resource.get("resourceType", "")
    requested_paths = fields.get(resource_type, []) + fields.get("*", [])
    if not requested_paths:
        return resource
    paths = list(dict.fromkeys(requested_paths + mandatory_fields))
    return _filter_by_tree(resource, _build_field_tree(paths))


def filter_bundle(
    bundle: Dict[str, Any], fields: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Apply filter_resource to every resource in a FHIR Bundle's entry list."""
    if not fields or "entry" not in bundle:
        return bundle
    entries = []
    for entry in bundle["entry"]:
        resource = entry.get("resource")
        if resource is not None:
            entry = {**entry, "resource": filter_resource(resource, fields)}
        entries.append(entry)
    return {**bundle, "entry": entries}


def filter_response(data: Any, fields: Dict[str, List[str]]) -> Any:
    """Route to filter_bundle or filter_resource depending on what the FHIR server returned."""
    if not fields or not isinstance(data, dict):
        return data
    if data.get("resourceType") == "Bundle":
        return filter_bundle(data, fields)
    return filter_resource(data, fields)


def fhirpath_filter_resource(
    resource: Dict[str, Any],
    fields: Dict[str, List[str]],
    preserve: List[str] = ["resourceType", "id"],
) -> Dict[str, Any]:
    """
    Filter a FHIR resource using FHIRPath expressions via fhirpathpy.
    fields is keyed by resourceType (or '*' for all types), with FHIRPath expressions
    relative to the resource type, e.g. ["name", "telecom.where(system='email')"].
    Returns a flat dict keyed by expression; does not reconstruct the original shape.
    """
    if not fields:
        return resource
    resource_type = resource.get("resourceType", "")
    paths = list(dict.fromkeys(fields.get(resource_type, []) + fields.get("*", [])))
    if not paths:
        return resource

    result: Dict[str, Any] = {}
    for field in preserve:
        val = resource.get(field)
        if val is not None:
            result[field] = val

    for expr in paths:
        if expr in preserve:
            continue
        path = expr if expr.startswith(resource_type) else f"{resource_type}.{expr}"
        values = fhircompile(path)(resource)
        if values:
            result[expr] = values[0] if len(values) == 1 else values

    return result


def fhirpath_filter_bundle(
    bundle: Dict[str, Any], fields: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Apply fhirpath_filter_resource to every resource in a FHIR Bundle's entry list."""
    if not fields or "entry" not in bundle:
        return bundle
    entries = []
    for entry in bundle["entry"]:
        resource = entry.get("resource")
        if resource is not None:
            entry = {**entry, "resource": fhirpath_filter_resource(resource, fields)}
        entries.append(entry)
    return {**bundle, "entry": entries}


def fhirpath_filter_response(data: Any, fields: Dict[str, List[str]]) -> Any:
    """Route to fhirpath_filter_bundle or fhirpath_filter_resource depending on what the FHIR server returned."""
    if not fields or not isinstance(data, dict):
        return data
    if data.get("resourceType") == "Bundle":
        return fhirpath_filter_bundle(data, fields)
    return fhirpath_filter_resource(data, fields)


async def create_async_fhir_client(
    config: ServerConfigs,
    access_token: str | None = None,
    extra_headers: dict | None = None,
) -> AsyncFHIRClient:
    """Create a FHIR AsyncClient with defaults."""

    client_kwargs: Dict = {
        "url": config.server_base_url,
        "aiohttp_config": {
            "timeout": aiohttp.ClientTimeout(total=config.mcp_request_timeout),
        },
        "extra_headers": extra_headers,
    }
    if access_token:
        client_kwargs["authorization"] = f"Bearer {access_token}"

    return AsyncFHIRClient(**client_kwargs)


async def get_bundle_entries(bundle: Dict[str, Any]) -> Dict[str, Any]:
    if bundle and "entry" in bundle and isinstance(bundle["entry"], list):
        logger.debug(f"found {len(bundle['entry'])} entries for type '{type}'")
        return {
            "entry": [
                entry.get("resource")
                for entry in bundle["entry"]
                if "resource" in entry
            ]
        }
    return bundle


def trim_resource_capabilities(
    capabilities: List[Dict[str, Any]],
) -> List[Dict[str, Optional[str]]]:
    logger.debug(
        f"trim_resource_capabilities called with {len(capabilities)} capabilities."
    )
    trimmed = [
        {
            "name": capability.get("name"),
            "documentation": capability.get("documentation"),
        }
        for capability in capabilities
        if "name" in capability or "documentation" in capability
    ]
    logger.debug(
        f"trim_resource_capabilities returning {len(trimmed)} trimmed capabilities."
    )
    return trimmed


async def get_operation_outcome_exception() -> dict:
    return await get_operation_outcome(
        code="exception", diagnostics="An unexpected internal error has occurred."
    )


async def get_operation_outcome_required_error(element: str = "") -> dict:
    return await get_operation_outcome(
        code="required", diagnostics=f"A required element {element} is missing."
    )


async def get_operation_outcome(
    code: str, diagnostics: str, severity: str = "error"
) -> dict:
    return {
        "resourceType": "OperationOutcome",
        "issue": [
            {
                "severity": severity,
                "code": code,
                "diagnostics": diagnostics,
            }
        ],
    }


async def get_capability_statement(metadata_url: str) -> Dict[str, Any]:
    """
    Discover CapabilityStatement from server's metadata endpoint.
    """
    try:
        logger.debug(f"Fetching CapabilityStatement from {metadata_url}")
        async with create_mcp_http_client() as client:
            response = await client.get(url=metadata_url, headers=get_default_headers())
            response.raise_for_status()
            metadata_json = response.json()
            logger.debug(f"OAuth metadata discovered: {metadata_json}")
            return metadata_json
    except Exception as ex:
        logger.exception(
            "Unable to invoke the FHIR metadata endpoint. Caused by, ", exc_info=ex
        )
        raise ValueError("Unable to fetch FHIR metadata")


def get_default_headers() -> Dict[str, str]:
    return {"Accept": "application/fhir+json", "Content-Type": "application/fhir+json"}


def build_user_profile(resource: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build user profile dictionary from FHIR resource.

    Args:
        resource: The FHIR resource dictionary of the user.

    Returns:
        Dict containing only mandatory user fields
    """

    # Define fields to extract from the resource
    fields_to_extract = [
        "id",
        "resourceType",
        "name",
        "gender",
        "birthDate",
        "telecom",
        "address",
    ]

    profile: Dict[str, Any] = {}
    # Add fields only if they exist and have values
    for field in fields_to_extract:
        value = resource.get(field)
        if value is not None:
            profile[field] = value

    return profile
