
# BatchGrids API Specification (Stub)

This is a living document. Add full OpenAPI schemas as endpoints are finalized.

## Current endpoints (planned)
### Health
- `GET /health`

### Images
- `POST /images/upload`
- `POST /images/{id}/process`
- `GET /images/{id}`

### Tools
- `POST /tools`
- `GET /tools`
- `GET /tools/{id}`
- `PATCH /tools/{id}`

### Bins
- `POST /bins`
- `GET /bins/{id}`
- `POST /bins/{id}/autofill`
- `POST /bins/{id}/export`

### Drawers
- `POST /drawers`
- `GET /drawers/{id}`
- `PATCH /bins/{id}` (assign drawer/tile)

### Lookup
- `GET /lookup?query=<text>`

### Exports
- `GET /bins/{id}/exports`
