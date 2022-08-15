from pipeline.schemas.user import UserCreate


def test_user_create_optional_username():
    none_username_payload = dict(
        email="email@company.com", password="ExamplePass123", username=None
    )
    UserCreate(**none_username_payload)
    assert True
