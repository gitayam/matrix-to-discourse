#ROADMAP.md 
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
- [x] Configuration of Post Title Generation.
    - [x] Allow users to configure the title generation GPT Prompt.
    - [x] Allow users to disable GPT title generation.
    - [x] Fallback to a default title if GPT fails.
- [x] Configuration of Trigger Words.
    - [x] Allow users to configure trigger words that will trigger the bot to create a post.
    - [x] Allow users to configure trigger words that will trigger the bot to search for a post.
- [x] Create a post based on links posted in the chat. : URL-handling-single-file
    - [x] List of URLs to trigger REGEX conversion and posting.
    - [x] Option to reply to any message containing a link and go through this process even if the link isn't in the list.
    - [x] REGEX to convert links to 12ft.io and archive.org links to bypass paywalls.
    - [x] Tag post with "posted-link" tag.
    - [x] Search for duplicate posts with the same URL, don't create a new post if it exists.
    - [x] Scrape the post and summarize it using GPT-4o-mini.
    - [x] Add the summary as a post body.
    - [x] Add the 12ft.io and archive.org links to the post body.
    - [x] Return the post link to the chat.
- [x] Relay Matrix messages to Discourse as a post reply. 

## Roadmap

- [ ] Enhanced Post Summarization Functionality.(working on) (Branch: Multiple-Message-Summaries) 
    - [ ] Implement `!fpost -n <number>` to summarize the last `<number>` messages into a post.
    - [ ] Implement `!fpost -h <hours>`, `-m <minutes>`, `-d <days>` to summarize messages from a specified timeframe.
    - [ ] Handle situations where the chat history does not go back as far as requested by defaulting to the available range.
    - [ ] Ensure functionality in disappearing chats by defaulting to the latest available messages up to the specified timeframe.
    - [ ] Allow More messages as context for post generation (Working on)
    - [ ] Handle media messages from the bridge.
        - Currently, Signal media is clumped together on Signal so a user response is to the first media message, not to the text body.
    - [ ] Handle multiple messages in a single reply.
    - [ ] Post Generation Improvements, GPT for body of the post as well using context.
- [ ] White-listed users can create posts in a specific category.
    - [ ] Based on the user's role in the Matrix room.
    - [ ] Based on User ID.
    - [ ] Based on homeserver.
- [ ] Direct message support
    - [ ] Direct message user with the post link as well
