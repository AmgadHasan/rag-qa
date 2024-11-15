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

QUESTIONS_SYSTEM_MESSAGE = """You're an excellent teacher that writes asks good questions. Return the output directly in markdown without an introduction."""
QUESTIONS_USER_MESSAGE = dedent('''\
    Write a list of questions for the requested topic based on the provided context and the requested question type.
    ## CONTEXT:
    """
    {context}
    """
    
    ## TOPIC:
    """
    {topic}
    """

    ## TYPE:
    """
    {type}
    """
''')
