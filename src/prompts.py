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

QUESTIONS_SYSTEM_MESSAGE = """You're an excellent teacher that writes asks good questions. Return a valid JSON that is a list of strings"""
QUESTIONS_USER_MESSAGE = dedent('''\
    Write a list of questions for the requested topic based on the provided context.
    The questions should be of type {type}.
    ## CONTEXT:
    """
    {context}
    """
    
    ## TOPIC:
    """
    {topic}
    """
''')
