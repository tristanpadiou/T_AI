from pydantic_ai.mcp import MCPServerStreamableHTTP, MCPServerSSE, MCPServerStdio

class MCP_server_helper:
    """
    A helper class to add MCP servers to the list of MCP servers
    **Example:**
    ```python
        mcp_server_helper=MCP_server_helper()
        mcp_server_helper.add_mcp_server(type='http', mcp_server_url='https://mcp.notion.com/mcp', headers=None)
        mcp_server_helper.add_mcp_server(type='sse', mcp_server_url='https://mcp.notion.com/sse', headers=None)
        mcp_server_helper.add_mcp_server(type='stdio', command='npx', args=['-y', 'mcp-remote', 'https://mcp.notion.com/mcp'], env=None)
        mcp_server_helper.get_mcp_servers()
    ```
    Returns:
        list: A list of MCP servers
    """
    def __init__(self):
        self.mcp_servers=[]
    def add_mcp_server(self, type:str, mcp_server_url:str = None, headers:dict = None, command:str = None, args:list = None, env:dict = None, timeout:int = 60):
        """
        Add an MCP server to the list of MCP servers
        Args:
            type: The type of MCP server to add, can be 'http', 'sse', or 'stdio'
            mcp_server_url: The URL of the MCP server to add
            headers: The headers to add to the MCP server
            command: The command to add to the MCP server
            args: The arguments to add to the MCP server
            env: The environment variables to add to the MCP server
            timeout: The timeout for the MCP server
        Example:
            ```python
            add_mcp_server(type='http', mcp_server_url='https://mcp.notion.com/mcp', headers=None)
            add_mcp_server(type='sse', mcp_server_url='https://mcp.notion.com/sse', headers=None)
            add_mcp_server(type='stdio', command='npx', args=['-y', 'mcp-remote', 'https://mcp.notion.com/mcp'], env=None)
            ```
        """
        if type == 'http':
            if headers is not None:
                self.mcp_servers.append(MCPServerStreamableHTTP(url=mcp_server_url, headers=headers, timeout=timeout))
            else:
                self.mcp_servers.append(MCPServerStreamableHTTP(url=mcp_server_url, timeout=timeout))
        elif type == 'sse':
            if headers is not None:
                self.mcp_servers.append(MCPServerSSE(url=mcp_server_url, headers=headers, timeout=timeout))
            else:
                self.mcp_servers.append(MCPServerSSE(url=mcp_server_url, timeout=timeout))
        elif type == 'stdio':
            if env is not None:
                self.mcp_servers.append(MCPServerStdio(command=command, args=args, env=env, timeout=timeout))
            else:
                self.mcp_servers.append(MCPServerStdio(command=command, args=args, timeout=timeout))
        else:
            raise ValueError(f"Invalid type: {type}")
    def get_mcp_servers(self):
        return self.mcp_servers
