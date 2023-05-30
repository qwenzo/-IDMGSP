# ChatGPT papers generator

## System Requirements
- [Nodejs](https://nodejs.org/en/) `version >= 18`

## Dependencies
- `chatgpt`
- `d3`
- `dotenv-safe`
- `puppeteer`

## Installation
- install nodejs with version >= 18.
- run `npm install` to get all the dependencies.

## Usage
- have a `.env` file with the parameters `OPENAI_EMAIL=` and  `OPENAI_PASSWORD=`. See `.env.example` for an example.
- run `node main.js start=0 end=500 output="output.csv" input="chatgpt.csv" input_column="Title`
    - all the arguments are optional and their default values are `start=0 end=500 output="output0_500.csv" input="chatgpt.csv" input_column="title"`.
    - `start` parameter is used to get the starting index (inclusive) to read from in `chatgpt.csv`.
    - `end` parameter is used to get the last index (exclusive) to read from in `chatgpt.csv`.
    - `chatgpt.csv` has to have the column `Title` for titles. title column can change using the parameter `input_column`.
- the script can be run in parallel using multiple accounts. Simply run the same script with another output file and change your credintials.


