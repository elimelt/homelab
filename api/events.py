from typing import Literal, Optional, TypedDict, Union


class Location(TypedDict, total=False):
    country: str
    city: str
    lat: Optional[float]
    lon: Optional[float]


class Visitor(TypedDict):
    ip: str
    location: Location
    connected_at: str


class JoinEvent(TypedDict):
    type: Literal["join"]
    visitor: Visitor


class LeaveEvent(TypedDict):
    type: Literal["leave"]
    ip: str


class PingEvent(TypedDict):
    type: Literal["ping"]


# Discriminated union of all visitor events the client may receive
VisitorEvent = Union[JoinEvent, LeaveEvent, PingEvent]


# Chat events
class ChatMessageEvent(TypedDict):
    type: Literal["chat_message"]
    channel: str
    sender: str
    text: str
    timestamp: str


