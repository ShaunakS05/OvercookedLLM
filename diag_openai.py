import openai, os, time, json, sys

# 1) Locate your key
key = os.getenv("OPENAI_API_KEY")
if not key:
    try:
        key = json.load(open("utils/chatgpt/openai.json"))["access_token"]
    except Exception:
        print("❌ No API key found (env var or utils/chatgpt/openai.json)")
        sys.exit(1)

openai.api_key = key
print("Using openai-python", getattr(openai, "__version__", "unknown"))

# 2) Test a simple ping
t0 = time.time()
try:
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":"ping"}],
        timeout=20,           # for v1.x client
        request_timeout=20    # for v0.x client
    )
    print("✔ reply:", res.choices[0].message.content, f"({time.time()-t0:.1f}s)")
except Exception as e:
    print("❌ exception after", round(time.time()-t0,1), "s\n", e)
