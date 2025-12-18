import unittest


def test_completions(service_endpoint='http://localhost:6001/evaluation'):
    def _create_client(url):
        from openai import OpenAI

        return OpenAI(
            api_key='model_config.api_key',
            base_url=url,
        )

    client = _create_client(service_endpoint)
    if client:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            frequency_penalty=0,
            max_tokens=200,
            n=1,
            presence_penalty=0,
            response_format={"type": "text"},
            seed=0,
            stop=None,
            temperature=0,
            top_p=1,
            stream=False,
            extra_body=None,
        )
        text = completion.choices[0].message.content
        assert isinstance(text, str), "Expect response"
