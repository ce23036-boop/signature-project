# Signature verification project

## Security settings

The application uses demo access codes only when Streamlit secrets are not
configured. For deployment, add these values in the app's Streamlit Cloud
secrets settings:

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
verification history. History is stored as metadata in SQLite and is visible
only after entering `admin_code` in the sidebar.

Streamlit Community Cloud storage is ephemeral, so SQLite is suitable for
demonstration and testing only. Use an external managed database before
relying on the history for production or compliance purposes.