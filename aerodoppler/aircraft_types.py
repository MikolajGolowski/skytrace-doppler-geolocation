"""ICAO 8643 Aircraft Type Designator classification (Propeller vs Jet).

Mirrors the filtering methodology described in Colsto's video: because lookup
tables can have missing entries, users can either exclude all known jet types
(preserving unknown codes so real propeller targets aren't dropped) or require
known propeller/turboprop types strictly.
"""

from __future__ import annotations

from typing import Literal

PropulsionClass = Literal["JET", "PROP", "TURBOPROP", "HELI", "UNKNOWN"]

KNOWN_JET_TYPES: frozenset[str] = frozenset(
    {
        # Boeing
        "B703", "B712", "B721", "B722", "B732", "B733", "B734", "B735", "B736",
        "B737", "B738", "B739", "B37M", "B38M", "B39M", "B3XM", "B741", "B742",
        "B743", "B744", "B748", "B752", "B753", "B762", "B763", "B764", "B772",
        "B773", "B77L", "B77W", "B778", "B779", "B788", "B789", "B78X",
        # Airbus
        "A220", "BCS1", "BCS3", "A306", "A30B", "A310", "A318", "A319", "A320",
        "A321", "A19N", "A20N", "A21N", "A332", "A333", "A338", "A339", "A342",
        "A343", "A345", "A346", "A359", "A35K", "A388",
        # Embraer Regional Jets
        "E135", "E145", "E170", "E75L", "E75S", "E190", "E195", "E290", "E295",
        "E50P", "E55P", "E545", "E550", "E35L",
        # Bombardier / Canadair CRJ & Global / Challenger
        "CRJ1", "CRJ2", "CRJ7", "CRJ9", "CRJX", "CL30", "CL35", "CL60", "GLEX",
        "GL5T", "GL7T",
        # McDonnell Douglas / Fokker / BAE
        "MD11", "MD81", "MD82", "MD83", "MD87", "MD88", "MD90", "DC10", "B462",
        "B463", "RJ85", "RJ1H", "F70", "F100",
        # Business Jets (Cessna Citation, Gulfstream, Learjet, Dassault, Hawker, HondaJet)
        "C25A", "C25B", "C25C", "C25M", "C500", "C501", "C510", "C525", "C550",
        "C551", "C560", "C56X", "C650", "C680", "C68A", "C700", "C750",
        "GLF2", "GLF3", "GLF4", "GLF5", "GLF6", "G280", "GA5C", "GA6C", "GA7C",
        "LJ24", "LJ25", "LJ31", "LJ35", "LJ40", "LJ45", "LJ55", "LJ60", "LJ75",
        "FA10", "FA20", "FA50", "FA7X", "FA8X", "F2TH", "F900", "H25A", "H25B",
        "H25C", "HA4T", "HDJT", "BE40", "PRM1", "SF50", "EA50", "MU30", "GALX",
        "ASTR", "WW24", "SBR1", "PC24",
    }
)

KNOWN_PISTON_PROP_TYPES: frozenset[str] = frozenset(
    {
        # Cessna singles & twins
        "C140", "C150", "C152", "C170", "C172", "C175", "C177", "C180", "C182",
        "C185", "C195", "C205", "C206", "C207", "C210", "C310", "C320", "C337",
        "C340", "C401", "C402", "C404", "C414", "C421", "C72R", "C77R", "C82R",
        # Piper
        "J3", "PA11", "PA12", "PA18", "PA22", "PA23", "PA24", "PA27", "PA28",
        "P28A", "P28B", "P28R", "P28T", "PA30", "PA32", "P32R", "PA34", "PA38",
        "PA44", "PA46",
        # Beechcraft
        "BE23", "BE24", "BE33", "BE35", "BE36", "BE55", "BE58", "BE60", "BE76",
        # Cirrus / Diamond / Mooney / Grumman / Van's RV / Tecnam / Flight Design
        "SR20", "SR22", "S22T", "DA20", "DA40", "DA42", "DA62", "M20P", "M20T",
        "AA1", "AA5", "RV4", "RV6", "RV7", "RV8", "RV9", "RV10", "RV12", "RV14",
        "P208", "P06T", "FDCT", "CTLS", "SLG2", "SLG4", "COL3", "COL4", "TRIN",
        "TOBA", "TAMP", "BL8", "CH7A", "GC1", "LNC2", "LNC4", "VELO", "EZ",
        "B350", "AC11", "AC50", "AC68",
    }
)

KNOWN_TURBOPROP_TYPES: frozenset[str] = frozenset(
    {
        "C208", "C425", "C441", "PC6T", "PC12", "TBM7", "TBM8", "TBM9", "BE10",
        "BE20", "BE30", "BE9L", "BE9T", "B190", "DH8A", "DH8B", "DH8C", "DH8D",
        "DHC6", "AT43", "AT45", "AT46", "AT72", "AT75", "AT76", "SF34", "SB20",
        "E120", "SW3", "SW4", "PA31", "PAY1", "PAY2", "PAY3", "PAY4", "P46T",
        "M600", "MU2", "AC90", "AC95", "K350", "EVOT", "EPIC", "KODI", "C130",
        "C30J", "D328", "JS31", "JS32", "JS41", "P180",
    }
)

KNOWN_HELI_TYPES: frozenset[str] = frozenset(
    {
        "R22", "R44", "R66", "B06", "B206", "B407", "B412", "B429", "EC20",
        "EC30", "EC35", "EC45", "EC55", "AS50", "AS55", "S76", "S92", "H500",
        "MD52", "A109", "A139", "B47G", "H60",
    }
)


def classify_icao_type(type_code: str | None) -> PropulsionClass:
    """Classify an ICAO aircraft type designator into JET / PROP / TURBOPROP / HELI / UNKNOWN."""
    if not type_code:
        return "UNKNOWN"
    code = type_code.strip().upper()
    if not code:
        return "UNKNOWN"
    if code in KNOWN_JET_TYPES:
        return "JET"
    if code in KNOWN_PISTON_PROP_TYPES:
        return "PROP"
    if code in KNOWN_TURBOPROP_TYPES:
        return "TURBOPROP"
    if code in KNOWN_HELI_TYPES:
        return "HELI"
    return "UNKNOWN"


def should_keep_aircraft(
    type_code: str | None,
    mode: Literal["exclude_jets", "prop_only", "all"] = "exclude_jets",
) -> bool:
    """Apply aircraft type filtering.

    - 'exclude_jets': Colsto's strategy — drop known jets, keep props, turboprops,
      helicopters, and unknown codes so missing lookup entries don't drop candidates.
    - 'prop_only': keep only known piston prop and turboprop aircraft.
    - 'all': keep everything.
    """
    if mode == "all":
        return True
    cls = classify_icao_type(type_code)
    if mode == "exclude_jets":
        return cls != "JET"
    if mode == "prop_only":
        return cls in ("PROP", "TURBOPROP")
    raise ValueError(f"Unknown aircraft filter mode: {mode}")
