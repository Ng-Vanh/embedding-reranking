import json
import glob
folder_path = r"D:\Lab\exp\embedding-reranking\test_result\webarena-inf-result\gemini-wo_explore"

pattern = "result_*.json"
file_paths = glob.glob(f"{folder_path}/{pattern}")
for file_path in file_paths:
    print(f"Checking results in {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    step_passed = 0
    test_cases_pass = 0
    all_test_cases = 0
    all_steps = 0

    all_test_cases += len(data)

    for item in data:
        status = item.get("status")
        steps_executed = item.get("steps_executed", [])

        all_steps += len(steps_executed)

        testcase_pass = (
            status == "success"
            or all(
                step.get("status") != "failed"
                for step in steps_executed
            )
        )

        if testcase_pass:
            test_cases_pass += 1

        for step in steps_executed:
            if step.get("status") != "failed":
                step_passed += 1

    print(f"Total test cases: {all_test_cases}")
    print(f"Passed test cases: {test_cases_pass}")
    print(f"Total steps executed: {all_steps}")
    print(f"Passed steps: {step_passed}")
    print(f"Test case pass rate: {test_cases_pass / all_test_cases:.2%}")