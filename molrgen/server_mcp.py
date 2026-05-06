from molrgen.server_utils.server_app import create_app, create_mcp

app = create_app()
mcp = create_mcp(app)


if __name__ == "__main__":
    import uvicorn

    mcp.setup_server()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
