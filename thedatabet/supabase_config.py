from supabase import create_client

SUPABASE_URL = "https://uzzthlvvebagmuqtpsba.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV6enRobHZ2ZWJhZ211cXRwc2JhIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0OTgzOTA1MywiZXhwIjoyMDY1NDE1MDUzfQ.0iD8YXwjmMmEvBXCnwwaIDQFrTEuXbbnLgOlbX7uFfc"

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

