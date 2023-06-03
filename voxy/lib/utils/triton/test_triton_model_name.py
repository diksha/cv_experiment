import unittest

from lib.utils.triton.triton_model_name import triton_model_name


class TestTritonModelName(unittest.TestCase):
    def test_triton_model_name(self):
        # trunk-ignore(pylint/C0301)
        artifact_model_path = "artifacts_03_05_2023_americold_modesto_0011_cha/256037fe-697a-4484-ab48-8c81b6ea18e1.pt"
        # trunk-ignore(pylint/C0301)
        expected_triton_model_name = "artifacts_03_05_2023_americold_modesto_0011_cha_256037fe_697a_4484_ab48_8c81b6ea18e1_pt_2f96c2"

        actual_name = triton_model_name(artifact_model_path, False)

        self.assertEqual(actual_name, expected_triton_model_name)

    def test_triton_model_with_ensemble_name(self):
        # trunk-ignore(pylint/C0301)
        artifact_model_path = "artifacts_03_05_2023_americold_modesto_0011_cha/256037fe-697a-4484-ab48-8c81b6ea18e1.pt"
        # trunk-ignore(pylint/C0301)
        expected_triton_model_name = "artifacts_03_05_2023_americold_modesto_0011_cha_256037fe_697a_4484_ab48_8c81b6ea18e1_pt_2f96c2_ensemble"

        actual_name = triton_model_name(artifact_model_path, True)

        self.assertEqual(actual_name, expected_triton_model_name)
