# Matrix to Discourse Bot Roadmap
Thanks for checking out the roadmap for the Matrix to Discourse Bot! This document outlines the features that are currently in development and planned for the future.


## How to Contribute
1. Fork the repository.
2. Create a new branch.
3. Implement the feature or bug fix.
4. Create a pull request.

## Current Features
- [x] Create a post on the Discourse forum using the replied-to message's content.
- [x] Search the Discourse forum for the specified query.
- [x] User Configuration via Web Interface.
    - [x] Configuration of Discourse API Key and Username.
    - [x] Configuration of Discourse Base URL.
- [x] Sync Matrix room to Discourse Category.
    - [x] Configuration of Matrix to Discourse Topic Mapping.
    - [x] Configuration of Unsorted Category ID.
- [x] Generate a title based on the body of the message using GPT-4o-mini.
    - [x] Configuration of GPT API Key and Model.
    - [x] Configuration of GPT Max Tokens and Temperature.

## Roadmap
- [ ] White-listed users can create posts in a specific category.
    - [ ] Based on the user's role in the Matrix room.
    - [ ] Based on User ID.
    - [ ] Based on homeserver.
- [ ] Configuration of Trigger Words.
    - [ ] Allow users to configure trigger words that will trigger the bot to create a post.
    - [ ] Allow users to configure trigger words that will trigger the bot to search for a post.
- [ ] Configuration of Post Title Generation.
    - [ ] Allow users to configure the title generation GPT Prompt.
    - [ ] Allow users to disable GPT title generation.
    - [ ] Fallback to a default title if GPT fails.
- [ ] Direct message support
    - [ ] Direct message user with the post link as well
- [ ] Allow More messages as context for post generation (Working on)
    - [ ] Handle media messages from the bridge.
        - Currently, Signal media is clumped together on Signal so a user response is to the first media message, not to the text body.
    - [ ] Handle multiple messages in a single reply.
    - [ ] Post Generation Improvements, GPT for body of the post as well using context.
- [ ] Enhanced Post Summarization Functionality.
    - [ ] Implement `!fpost -n <number>` to summarize the last `<number>` messages into a post.
    - [ ] Implement `!fpost -h <hours>`, `-m <minutes>`, `-d <days>` to summarize messages from a specified timeframe.
    - [ ] Handle situations where the chat history does not go back as far as requested by defaulting to the available range.
    - [ ] Ensure functionality in disappearing chats by defaulting to the latest available messages up to the specified timeframe.
