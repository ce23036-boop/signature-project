# Signature verification project

## Security settings

The application requires access codes in Streamlit secrets. It does not
contain fallback or demo codes in the source code. Add these values in the
app's Streamlit Cloud secrets settings:

```toml
[users]
person_1 = "replace-with-a-private-code"
person_2 = "replace-with-a-private-code"
person_3 = "replace-with-a-private-code"
person_4 = "replace-with-a-private-code"
person_5 = "replace-with-a-private-code"

admin_code = "replace-with-a-private-admin-code"
```

The application never writes access codes or signature images to the
verification history. History is stored as metadata in Supabase and is visible
only after entering `admin_code` in the sidebar.

Add the Supabase server-side credentials to Streamlit Cloud Secrets:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SECRET_KEY = "your-server-side-secret-key"
```

Use a Supabase secret/server-side key, not a publishable key. Never commit
either value to GitHub.
