"""Image references shared by user input and tool results.

Only references are ever stored ({"ref", "detail"?}); bytes stay with the
host, and BaseAgent.resolve_image turns a reference into a URL right before a
model call.
"""

from dataclasses import dataclass

IMAGE_DETAILS = ("auto", "low", "high")


@dataclass(frozen=True)
class Image:
    """An image reference placed inside interleaved input, e.g. ["See:", Image("file:1")]."""

    ref: str
    detail: str | None = None

    def as_dict(self):
        return {"ref": self.ref} if self.detail is None else {"ref": self.ref, "detail": self.detail}


def normalize_images(images):
    """Validate images into stored references: [{"ref": str, "detail"?: str}].

    Each item is a non-empty reference string (host file ID, https URL or
    data URL), an Image, or a {"ref", "detail"} dictionary, where detail is
    auto, low or high and is forwarded to the model unchanged.
    """
    if images is None:
        return []
    if not isinstance(images, (list, tuple)):
        raise ValueError("images must be a list of references")
    result = []
    for image in images:
        item = ({"ref": image} if isinstance(image, str)
                else image.as_dict() if isinstance(image, Image) else image)
        if not isinstance(item, dict) or not set(item) <= {"ref", "detail"}:
            raise ValueError("An image must be a reference string or a {ref, detail} object")
        if not isinstance(item.get("ref"), str) or not item["ref"].strip():
            raise ValueError("Image ref must be a non-empty string")
        if "detail" in item and item["detail"] not in IMAGE_DETAILS:
            raise ValueError("Image detail must be auto, low or high")
        result.append(dict(item))
    return result


def split_interleaved(parts, normalize=normalize_images):
    """Split ["text", Image(...), ...] into stored text plus images placed by offset.

    Text segments are concatenated into content; each image records "at", the
    character offset in content where it appears, so the model sees text and
    images in the original order. Shared by user input and ToolResult.
    """
    if not isinstance(parts, (list, tuple)):
        raise ValueError("Interleaved content must be a list of text strings and Image objects")
    content, images = "", []
    for part in parts:
        if isinstance(part, str):
            content += part
        elif isinstance(part, Image):
            image = normalize([part])[0]
            image["at"] = len(content)
            images.append(image)
        else:
            raise ValueError("Interleaved input may only contain text strings and Image objects")
    if not images and not content.strip():
        raise ValueError("Interleaved input needs non-empty text or at least one image")
    return content, images
