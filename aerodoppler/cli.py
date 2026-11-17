"""Command-line interface for SkyTrace / AeroDoppler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adsb import load_adsb_path
from .dem import FlatDEM, GridDEM, OpenMeteoDEM
from .locator import EventQuery, run_geolocation_pipeline
from .physics import haversine_m
from .simulator import generate_demo_scenario
from .spectrogram import compute_spectrogram, detect_doppler_events, load_audio


def cmd_demo(args: argparse.Namespace) -> None:
    sc = generate_demo_scenario(preset=args.preset)
    spec = compute_spectrogram(sc.samples, sc.sample_rate)
    events = detect_doppler_events(spec, max_events=2 if sc.mode == "two_event" else 1)
    print(f"Scenario: {sc.name}")
    print(f"Detected {len(events)} Doppler event(s):")
    for i, ev in enumerate(events, start=1):
        print(f"  Event {i}: {ev.to_dict()}")

    if not events:
        print("No Doppler events detected.")
        return

    e1 = events[0]
    q1 = EventQuery(
        e1.t_cpa_s,
        e1.f0_hz,
        e1.speed_min_ms,
        e1.speed_max_ms,
        e1.d_min_m,
        e1.d_max_m,
        e1.t_tolerance_s,
        e1.f0_tolerance_hz,
    )
    q2 = None
    if sc.mode == "two_event" and len(events) >= 2:
        e2 = events[1]
        q2 = EventQuery(
            e2.t_cpa_s,
            e2.f0_hz,
            e2.speed_min_ms,
            e2.speed_max_ms,
            e2.d_min_m,
            e2.d_max_m,
            e2.t_tolerance_s,
            e2.f0_tolerance_hz,
        )

    rep = run_geolocation_pipeline(
        sc.tracks,
        spec,
        q1,
        q2,
        mode=sc.mode,  # type: ignore[arg-type]
        dem=sc.dem,
        known_utc_cpa1=sc.known_utc_cpa1,
    )
    print("\nPipeline Funnel:", json.dumps(rep.funnel, indent=2))
    if rep.top_candidates:
        top = rep.top_candidates[0]
        err = haversine_m(sc.true_lat, sc.true_lon, top.lat, top.lon)
        print(f"\nTop Candidate: lat={top.lat:.6f}, lon={top.lon:.6f}, score={top.total_score:.3f}")
        print(f"Flights: {top.flight1_callsign} ({top.flight1_type})" + (f" + {top.flight2_callsign} ({top.flight2_type})" if top.flight2_callsign else ""))
        print(f"Ground Truth:  lat={sc.true_lat:.6f}, lon={sc.true_lon:.6f}")
        print(f"Localization Error: {err:.1f} m")


def cmd_locate(args: argparse.Namespace) -> None:
    samples, sr = load_audio(args.audio, target_sr=2000)
    spec = compute_spectrogram(samples, sr)
    tracks = load_adsb_path(args.adsb)

    if args.dem == "open-meteo":
        dem = OpenMeteoDEM()
    elif args.dem and Path(args.dem).exists():
        p = Path(args.dem)
        dem = GridDEM.from_npz(p) if p.suffix == ".npz" else GridDEM.from_ascii_grid(p)
    else:
        dem = FlatDEM(0.0)

    if args.auto_detect or args.t1 is None:
        events = detect_doppler_events(spec, max_events=2 if args.mode == "two_event" else 1)
        if not events:
            raise SystemExit("No Doppler curves detected automatically. Specify --t1/--f1 manually.")
        e1 = events[0]
        q1 = EventQuery(e1.t_cpa_s, e1.f0_hz, e1.speed_min_ms, e1.speed_max_ms, e1.d_min_m, e1.d_max_m)
        q2 = None
        if args.mode == "two_event":
            if len(events) < 2:
                raise SystemExit("Two-event mode requested, but only 1 event detected.")
            e2 = events[1]
            q2 = EventQuery(e2.t_cpa_s, e2.f0_hz, e2.speed_min_ms, e2.speed_max_ms, e2.d_min_m, e2.d_max_m)
    else:
        q1 = EventQuery(args.t1, args.f1, args.v1_min, args.v1_max, args.d1_min, args.d1_max)
        q2 = (
            EventQuery(args.t2, args.f2, args.v2_min, args.v2_max, args.d2_min, args.d2_max)
            if args.mode == "two_event"
            else None
        )

    rep = run_geolocation_pipeline(
        tracks,
        spec,
        q1,
        q2,
        mode=args.mode,
        dem=dem,
        known_utc_cpa1=args.utc_time,
    )
    out = json.dumps(rep.to_dict(), indent=2)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"Report saved to {args.out}")
    else:
        print("Funnel:", json.dumps(rep.funnel, indent=2))
        for i, c in enumerate(rep.top_candidates[:5], start=1):
            print(f"#{i}: ({c.lat:.6f}, {c.lon:.6f}) score={c.total_score:.3f} flights={c.flight1_callsign}/{c.flight2_callsign}")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)


def main() -> None:
    parser = argparse.ArgumentParser(prog="aerodoppler", description="SkyTrace / AeroDoppler Acoustic Aircraft Geolocation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run built-in synthetic benchmark")
    p_demo.add_argument("--preset", default="quick_two_event", choices=["quick_two_event", "colsto_920s", "single_1min_faint"])
    p_demo.set_defaults(func=cmd_demo)

    p_loc = sub.add_parser("locate", help="Geolocate an audio/video file against ADS-B tracks")
    p_loc.add_argument("--audio", required=True, help="Path to audio or video file")
    p_loc.add_argument("--adsb", required=True, help="Path to ADS-B tracks (.json, .csv, .parquet, or readsb directory)")
    p_loc.add_argument("--mode", default="two_event", choices=["two_event", "single_event"])
    p_loc.add_argument("--dem", default="flat", help="'flat', 'open-meteo', or path to .npz/.asc DEM")
    p_loc.add_argument("--auto-detect", action="store_true", help="Automatically detect Doppler events")
    p_loc.add_argument("--t1", type=float)
    p_loc.add_argument("--f1", type=float, default=130.0)
    p_loc.add_argument("--v1-min", type=float, default=45.0)
    p_loc.add_argument("--v1-max", type=float, default=120.0)
    p_loc.add_argument("--d1-min", type=float, default=200.0)
    p_loc.add_argument("--d1-max", type=float, default=4500.0)
    p_loc.add_argument("--t2", type=float)
    p_loc.add_argument("--f2", type=float, default=115.0)
    p_loc.add_argument("--v2-min", type=float, default=45.0)
    p_loc.add_argument("--v2-max", type=float, default=120.0)
    p_loc.add_argument("--d2-min", type=float, default=200.0)
    p_loc.add_argument("--d2-max", type=float, default=4500.0)
    p_loc.add_argument("--utc-time", type=float, help="Known UTC epoch of Event 1 CPA (for single_event mode)")
    p_loc.add_argument("--out", help="Output JSON report path")
    p_loc.set_defaults(func=cmd_locate)

    p_srv = sub.add_parser("serve", help="Launch FastAPI Web UI")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8080)
    p_srv.add_argument("--reload", action="store_true")
    p_srv.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
