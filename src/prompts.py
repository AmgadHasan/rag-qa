from textwrap import dedent

SUMMARY_SYSTEM_MESSAGE = (
    """You're an excellent teacher that writes excellent summaries."""
)
SUMMARY_USER_MESSAGE = dedent('''\
    Write a summary for the requested topic based on the provided context.
    
    ## CONTEXT:
    """
    {context}
    """
    
    ## TOPIC:
    """
    {topic}
    """
''')
