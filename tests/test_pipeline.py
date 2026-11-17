"""Comprehensive unit and end-to-end integration tests for SkyTrace / AeroDoppler."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from aerodoppler.adsb import FlightTrack, load_adsb_path
from aerodoppler.aircraft_types import classify_icao_type, should_keep_aircraft
from aerodoppler.countermeasures import evaluate_countermeasure
from aerodoppler.dem import FlatDEM, SyntheticTerrainDEM
from aerodoppler.locator import EventQuery, run_geolocation_pipeline
from aerodoppler.physics import (
    DEFAULT_SPEED_OF_SOUND,
    compute_track_cpa,
    ecef_to_geodetic,
    enu_to_geodetic,
    geodetic_to_ecef,
    geodetic_to_enu,
    haversine_m,
    ideal_doppler_curve,
    model_doppler_naive,
    model_doppler_retarded,
)
from aerodoppler.simulator import generate_demo_scenario
from aerodoppler.spectrogram import compute_spectrogram, detect_doppler_events
from server import app


class TestPhysicsAndCoordinates(unittest.TestCase):
    def test_wgs84_ecef_enu_roundtrip(self) -> None:
        lat, lon, alt = 40.8724, -73.5038, 65.2
        ecef = geodetic_to_ecef(lat, lon, alt)
        la2, lo2, al2 = ecef_to_geodetic(ecef)
        self.assertAlmostEqual(float(la2), lat, places=6)
        self.assertAlmostEqual(float(lo2), lon, places=6)
        self.assertAlmostEqual(float(al2), alt, places=2)

        enu = geodetic_to_enu(lat + 0.01, lon - 0.01, alt + 20.0, lat, lon, alt)
        la3, lo3, al3 = enu_to_geodetic(enu, lat, lon, alt)
        self.assertAlmostEqual(float(la3), lat + 0.01, places=6)
        self.assertAlmostEqual(float(lo3), lon - 0.01, places=6)
        self.assertAlmostEqual(float(al3), alt + 20.0, places=2)

    def test_retarded_time_matches_ideal_linear_motion(self) -> None:
        # Constant linear flyover at 80 m/s, 1000 m altitude directly overhead
        lat0, lon0 = 40.87, -73.50
        t_arr = np.linspace(1000.0, 1120.0, 241)
        e_arr = (t_arr - 1060.0) * 80.0
        n_arr = np.zeros_like(e_arr)
        u_arr = np.full_like(e_arr, 1000.0)
        lats, lons, alts = enu_to_geodetic(np.column_stack([e_arr, n_arr, u_arr]), lat0, lon0, 0.0)
        tr = FlightTrack("test01", "N12345", "C172", t_arr, lats, lons, alts)

        obs_ecef = geodetic_to_ecef(lat0, lon0, 0.0)
        t_cpa, d_cpa, t_del = compute_track_cpa(tr.timestamps, tr.pos_ecef, tr.vel_ecef, obs_ecef)
        self.assertAlmostEqual(t_cpa, 1060.0, delta=0.2)
        self.assertAlmostEqual(d_cpa, 1000.0, delta=2.0)
        self.assertAlmostEqual(t_del, 1060.0 + 1000.0 / DEFAULT_SPEED_OF_SOUND, delta=0.2)

        t_eval = np.linspace(1035.0, 1090.0, 40)
        f_ret, valid = model_doppler_retarded(tr.timestamps, tr.pos_ecef, tr.vel_ecef, obs_ecef, t_eval, f0=130.0)
        f_ideal = ideal_doppler_curve(t_eval[valid], f0=130.0, speed_ms=80.0, d_cpa_m=1000.0, t_cpa_delayed=t_del)
        max_diff = float(np.max(np.abs(f_ret[valid] - f_ideal)))
        self.assertLess(max_diff, 0.25)

        # Verify naive (unretarded) diverges significantly around CPA
        f_nai, _ = model_doppler_naive(tr.timestamps, tr.pos_ecef, tr.vel_ecef, obs_ecef, t_eval, f0=130.0)
        naive_diff = float(np.max(np.abs(f_nai[valid] - f_ideal)))
        self.assertGreater(naive_diff, 4.0)


class TestICAOAndADSB(unittest.TestCase):
    def test_icao_classification_colsto_filter(self) -> None:
        self.assertEqual(classify_icao_type("B738"), "JET")
        self.assertEqual(classify_icao_type("A320"), "JET")
        self.assertEqual(classify_icao_type("C182"), "PROP")
        self.assertEqual(classify_icao_type("PC12"), "TURBOPROP")
        self.assertFalse(should_keep_aircraft("B738", "exclude_jets"))
        self.assertTrue(should_keep_aircraft("C182", "exclude_jets"))
        # Unknown codes are kept in exclude_jets mode so missing lookup entries aren't lost
        self.assertTrue(should_keep_aircraft("ZZZZ", "exclude_jets"))
        self.assertFalse(should_keep_aircraft("ZZZZ", "prop_only"))

    def test_readsb_trace_gz_loader(self) -> None:
        payload = {
            "icao": "a12345",
            "t": "C172",
            "r": "N172SP",
            "timestamp": 1725800000.0,
            "trace": [
                [0.0, 40.86, -73.52, 3000],
                [10.0, 40.87, -73.51, 3050],
                [20.0, 40.88, -73.50, 3100],
                [30.0, 40.89, -73.49, 3150],
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "a12345.json.gz"
            with gzip.open(p, "wt", encoding="utf-8") as f:
                json.dump(payload, f)
            tracks = load_adsb_path(p)
            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0].type_code, "C172")
            self.assertEqual(tracks[0].propulsion_class, "PROP")
            self.assertGreater(tracks[0].median_speed_ms, 50.0)


class TestGeolocationPipelines(unittest.TestCase):
    def test_two_event_end_to_end_accuracy_and_silence_filter(self) -> None:
        sc = generate_demo_scenario("quick_two_event", seed=42)
        spec = compute_spectrogram(sc.samples, sc.sample_rate)
        events = detect_doppler_events(spec, max_events=2)
        self.assertEqual(len(events), 2)

        e1, e2 = events[0], events[1]
        q1 = EventQuery(e1.t_cpa_s, e1.f0_hz, e1.speed_min_ms, e1.speed_max_ms, e1.d_min_m, e1.d_max_m, e1.t_tolerance_s, e1.f0_tolerance_hz)
        q2 = EventQuery(e2.t_cpa_s, e2.f0_hz, e2.speed_min_ms, e2.speed_max_ms, e2.d_min_m, e2.d_max_m, e2.t_tolerance_s, e2.f0_tolerance_hz)

        rep_no_silence = run_geolocation_pipeline(
            sc.tracks, spec, q1, q2, mode="two_event", dem=sc.dem, enable_silence_filter=False
        )
        rep_with_silence = run_geolocation_pipeline(
            sc.tracks, spec, q1, q2, mode="two_event", dem=sc.dem, enable_silence_filter=True
        )

        # Silence filter must eliminate decoy airport points
        self.assertLess(
            rep_with_silence.funnel["after_silence_filter"],
            rep_no_silence.funnel["after_time_offset"],
        )
        self.assertTrue(len(rep_with_silence.top_candidates) >= 1)
        top = rep_with_silence.top_candidates[0]
        self.assertEqual(top.flight1_callsign, "N418CT")
        self.assertEqual(top.flight2_callsign, "N920PA")
        err_m = haversine_m(sc.true_lat, sc.true_lon, top.lat, top.lon)
        self.assertLess(err_m, 120.0)

    def test_single_event_1min_faint_traced(self) -> None:
        sc = generate_demo_scenario("single_1min_faint", seed=7)
        spec = compute_spectrogram(sc.samples, sc.sample_rate)
        events = detect_doppler_events(spec, max_events=1)
        self.assertEqual(len(events), 1)
        ev = events[0]

        obs_ecef = geodetic_to_ecef(sc.true_lat, sc.true_lon, sc.true_elev_m)
        t_pts = np.linspace(max(3.0, ev.t_cpa_s - 18.0), min(57.0, ev.t_cpa_s + 18.0), 12)
        f_pts, _ = model_doppler_retarded(
            sc.tracks[0].timestamps,
            sc.tracks[0].pos_ecef,
            sc.tracks[0].vel_ecef,
            obs_ecef,
            sc.recording_start_utc + t_pts,
            sc.true_f0_1,
        )
        traced = [(float(tt), float(ff)) for tt, ff in zip(t_pts, f_pts) if np.isfinite(ff)]
        q1 = EventQuery(
            ev.t_cpa_s,
            ev.f0_hz,
            ev.speed_min_ms,
            ev.speed_max_ms,
            ev.d_min_m,
            ev.d_max_m,
            t_tolerance_s=2.0,
            traced_points=traced,
        )
        rep = run_geolocation_pipeline(
            sc.tracks,
            spec,
            q1,
            mode="single_event",
            dem=sc.dem,
            known_utc_cpa1=sc.known_utc_cpa1,
            known_utc_tolerance_s=20.0,
        )
        top = rep.top_candidates[0]
        self.assertEqual(top.flight1_callsign, "N718SR")
        err_m = haversine_m(sc.true_lat, sc.true_lon, top.lat, top.lon)
        self.assertLess(err_m, 250.0)

    def test_countermeasures_highpass_fails_while_steep_bandstop_mitigates(self) -> None:
        sc = generate_demo_scenario("quick_two_event", seed=42)
        res_hp = evaluate_countermeasure(
            sc.samples, sc.sample_rate, sc.true_event1_audio_s, sc.true_f0_1, 78.0, 1000.0, "highpass_boost", 150.0
        )
        self.assertTrue(res_hp["curve_still_detectable"])
        res_steep = evaluate_countermeasure(
            sc.samples, sc.sample_rate, sc.true_event1_audio_s, sc.true_f0_1, 78.0, 1000.0, "steep_bandstop", 450.0
        )
        self.assertFalse(res_steep["curve_still_detectable"])


class TestFastAPIServer(unittest.TestCase):
    def test_api_end_to_end(self) -> None:
        client = TestClient(app)
        r_health = client.get("/api/health")
        self.assertEqual(r_health.status_code, 200)

        r_demo = client.post("/api/demo/load", json={"preset": "quick_two_event"})
        self.assertEqual(r_demo.status_code, 200)
        demo_data = r_demo.json()
        self.assertEqual(len(demo_data["detected_events"]), 2)
        ev1, ev2 = demo_data["detected_events"]

        r_loc = client.post(
            "/api/locate",
            json={
                "mode": "two_event",
                "event1": ev1,
                "event2": ev2,
            },
        )
        self.assertEqual(r_loc.status_code, 200)
        loc_data = r_loc.json()
        self.assertIn("ground_truth_eval", loc_data)
        self.assertLess(loc_data["ground_truth_eval"]["error_m"], 120.0)

        r_cm = client.post(
            "/api/countermeasures/test",
            json={
                "t_cpa_s": ev1["t_cpa_s"],
                "f0_hz": ev1["f0_hz"],
                "speed_ms": ev1["speed_ms"],
                "d_cpa_m": ev1["d_cpa_m"],
                "method": "highpass_boost",
                "param": 160.0,
            },
        )
        self.assertEqual(r_cm.status_code, 200)
        self.assertIn("curve_retention_pct", r_cm.json())


if __name__ == "__main__":
    unittest.main()
