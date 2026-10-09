from config.runtime_profiles import all_runtime_profiles


def test_six_concrete_runtime_profiles_are_configured():
    profiles = all_runtime_profiles()
    assert len(profiles) == 6
    assert {profile.id for profile in profiles} == {
        "django-sqlite-py312",
        "fastapi-py312",
        "flask-py312",
        "express-node20",
        "react-node20",
        "springboot-java21",
    }

