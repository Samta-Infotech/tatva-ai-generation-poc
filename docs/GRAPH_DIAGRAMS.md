# LangGraph-Style Diagrams

## PBQ

```text
START
 |
 v
load_runtime_profile
 |
 v
analyze_pbq_requirements
 |
 v
design_pbq
 |
 v
validate_pbq_design
 |
 +---- FAIL ---> repair_pbq_design ---+
 |                                    |
 +<-----------------------------------+
 |
 PASS
 |
 v
generate_candidate_workspace
 |
 v
generate_reference_workspace
 |
 v
generate_tests
 |
 v
materialize_reference_workspace
 |
 v
execute_reference
 |
 +---- FAIL ---> repair_reference
 |
 PASS
 |
 v
validate_test_strength
 |
 +---- WEAK ---> repair_tests
 |
 PASS
 |
 v
validate_pbq_difficulty
 |
 +---- FAIL ---> repair_pbq_design
 |
 PASS
 |
 v
finalize_pbq
 |
 END
```

## MCQ

```text
START
 |
 v
analyze_mcq_request
 |
 v
create_difficulty_blueprint
 |
 v
generate_mcq
 |
 v
validate_mcq_structure
 |
 v
independent_solver
 |
 v
validate_answer
 |
 v
validate_distractors
 |
 v
validate_difficulty
 |
 +---- FAIL ---> repair_mcq ---+
 |                             |
 +<----------------------------+
 |
 PASS
 |
 v
finalize_mcq
 |
 END
```

