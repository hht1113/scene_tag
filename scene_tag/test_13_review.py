import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("13_review.py")
spec = importlib.util.spec_from_file_location("review13", MODULE_PATH)
review13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review13)


class ReviewPlatformTests(unittest.TestCase):
    def test_only_groups_one_to_seven_are_configured(self):
        self.assertTrue(review13.LABEL_CATEGORIES)
        self.assertTrue(
            all(key.startswith(review13.ALLOWED_MAJOR_PREFIXES) for key in review13.LABEL_CATEGORIES)
        )
        self.assertNotIn("08_车道接近", review13.LABEL_CATEGORIES)
        self.assertNotIn("09_汇入分流", review13.LABEL_CATEGORIES)
        self.assertNotIn("00_其他", review13.LABEL_CATEGORIES)

    def test_parse_structured_label_filters_non_target_groups(self):
        structured = {
            "tags": [
                {
                    "major_category": "01_DynamicInteraction",
                    "sub_category": "前车急刹",
                    "sub_category_en": "DynamicInteraction_LeadVehicleSuddenBrake",
                    "time_evidence": [{"start": 1.0, "end": 3.0}],
                    "confidence": {"sub": 0.9},
                },
                {
                    "major_category": "08_LaneApproach",
                    "sub_category": "相邻车辆接近",
                    "sub_category_en": "LaneApproach_AdjacentVehicle",
                    "time_evidence": [{"start": 3.0, "end": 5.0}],
                    "confidence": {"sub": 0.8},
                },
                {
                    "major_category": "00_Other",
                    "sub_category": "其他",
                    "sub_category_en": "else",
                    "time_evidence": [{"start": 5.0, "end": 7.0}],
                    "confidence": {"sub": 0.7},
                },
            ]
        }
        segs = review13._parse_structured_label(structured)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["label"], "DynamicInteraction_LeadVehicleSuddenBrake")

    def test_normalize_filters_existing_segments(self):
        anns = [
            {
                "video_path": "/tmp/a.mp4",
                "segments": [
                    {"label": "LaneCruising_RoadSpeedLimit", "start": 0, "end": 20, "confidence": 90},
                    {"label": "LaneApproach_AdjacentVehicle", "start": 0, "end": 10, "confidence": 60},
                    {"label": "else", "start": 10, "end": 20, "confidence": 50},
                ],
            }
        ]
        out = review13._normalize_annotations(anns)
        self.assertEqual(len(out[0]["segments"]), 1)
        self.assertEqual(out[0]["segments"][0]["label"], "LaneCruising_RoadSpeedLimit")


    def test_normalize_supports_tags_field(self):
        anns = [
            {
                "video_path": "/tmp/b.mp4",
                "tags": [
                    {
                        "major_category": "05_LaneCruising",
                        "sub_category": "稳态跟车",
                        "sub_category_en": "LaneCruising_SteadyFollowing",
                        "time_evidence": [{"start": 0.0, "end": 20.0}],
                        "confidence": {"major": 0.95, "sub": 0.9},
                    },
                    {
                        "major_category": "08_LaneApproach",
                        "sub_category": "相邻VRU接近",
                        "sub_category_en": "LaneApproach_AdjacentVRU",
                        "time_evidence": [{"start": 5.0, "end": 10.0}],
                        "confidence": {"sub": 0.7},
                    },
                ],
            }
        ]
        out = review13._normalize_annotations(anns)
        self.assertEqual(len(out[0]["segments"]), 1)
        self.assertEqual(out[0]["segments"][0]["label"], "LaneCruising_SteadyFollowing")
        self.assertEqual(out[0]["segments"][0]["start"], 0.0)

    def test_normalize_tags_as_list_directly(self):
        anns = [
            {
                "video_path": "/tmp/c.mp4",
                "tags": [
                    {
                        "major_category": "02_TrafficLight",
                        "sub_category": "直行红绿灯起停",
                        "sub_category_en": "TrafficLight_StraightStopOrGo",
                        "time_evidence": [{"start": 2.0, "end": 8.0}],
                        "confidence": {"sub": 0.85},
                    },
                ],
            }
        ]
        out = review13._normalize_annotations(anns)
        self.assertEqual(len(out[0]["segments"]), 1)
        self.assertEqual(out[0]["segments"][0]["label"], "TrafficLight_StraightStopOrGo")


if __name__ == "__main__":
    unittest.main()
