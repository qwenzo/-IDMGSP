import { ChatGPTAPIBrowser } from 'chatgpt'
import * as fs from "fs"
import * as d3 from "d3"
import dotenv from 'dotenv-safe'
import chalk from 'chalk';

dotenv.config()

// node main.js start=0 end=500

// use puppeteer to bypass cloudflare (headful because of captchas)
const api = new ChatGPTAPIBrowser({
  email: process.env.OPENAI_EMAIL,
  password: process.env.OPENAI_PASSWORD
})

await api.initSession()

async function prompt(text, options) {
  const result = await api.sendMessage(text, options)
  return result.response
}

function get_skipped_titles(file = "skipped.csv") {
  var skipped_titles = []
  
  try {
    console.log(`reading ${file}`)
    const texOutput =  fs.readFileSync(file, {encoding:'utf8', flag:'r'})
    skipped_titles = d3.csvParse(texOutput).map(row => row["title"]);
    console.log(skipped_titles)
    console.log(`${skipped_titles.length} titles are skipped.\n`);
  } catch (err) {
    if (err.code === 'ENOENT') {
      console.log(`${file} is missing writing a new one.\n`);
      fs.writeFileSync(file, "id, title\n")
    }
  }

  return skipped_titles
}

(async function () {
  var START = 0
  var END = 500
  var OUTPUT_IN = ""
  var INPUT = "chatgpt.csv"
  var INPUT_CLMN = "title"
  const TRIES = 3

  process.argv.slice(2).forEach(function (txt, index) {
    const obj = txt.split("=")
    const attr = obj[0]
    const val = obj[1]

    if (attr == "start")
      START = Number(val)
    if (attr == "end")
      END = Number(val)
    if (attr == "output")
      OUTPUT_IN = val
    if (attr == "input")
      INPUT = val
    if (attr == "input_column")
      INPUT_CLMN = val
  });

  const OUTPUT = OUTPUT_IN || `output${START}_${END}.csv`

  // read csv
  var data = []
  var ids = []

  try {
    console.log(`reading ${INPUT} with start ${START} (inclusive) and end ${END} (exclusive)`)
    const textInput =  fs.readFileSync(INPUT, {encoding:'utf8', flag:'r'})
    data = d3.csvParse(textInput).map(row => row[INPUT_CLMN]);
    data = data.slice(START, END)

    ids = d3.csvParse(textInput).map(row => row["id"]);
    ids = ids.slice(START, END)
  } catch (err) {
    console.log(err)
  }

  const options = {
    // onProgress: (res) => {
    //   console.log(res.response)
    // }
  }

  // continue pasrsing.
  var found_titles = []

  try {
    console.log(`reading ${OUTPUT}`)
    const texOutput =  fs.readFileSync(OUTPUT, {encoding:'utf8', flag:'r'})
    found_titles = d3.csvParse(texOutput).map(row => row["title"]);
    console.log(`found ${found_titles.length} titles.\n`);
  } catch (err) {
    if (err.code === 'ENOENT') {
      console.log(`${OUTPUT} is missing writing a new one.\n`);
      fs.writeFileSync(OUTPUT, "title,abstract,introduction,conclusion,skipped\n")
    }
  }

  var skipped_titles = get_skipped_titles()

  console.log(skipped_titles)

  // create promise timer.
  const timer = ms => new Promise(res => setTimeout(res, ms))

  const CHATGPTTIME = 3000 // in ms

  for (var [index, title] of data.entries()) {
    if (!found_titles.includes(title) && !skipped_titles.includes(title)){
      const title_preprocessed = title.trim().replace(/"/g, '""')
      console.log(`title number ${index + 1} out of ${data.length}`)
      const prompt_input = `Write a document with the title '${title}' with an abstract, an introduction and a conclusion.`
      var text = `prompt: ${prompt_input}`
      console.log(text)

      var matches = null
      var tries_cnt = TRIES

      while(!matches && tries_cnt > 0) {
        const regex = /Abstract:([\s\S]*)?Introduction:([\s\S]*)?Conclusion:([\s\S]*)?/g
        console.log("recieving chatgpt output...")
        const result = await prompt(prompt_input, options)
        console.log(`${result}\n`)

        const trial_txt = `Trial ${TRIES - tries_cnt} of extracting relevant data from chatgpt output.`
        console.log(trial_txt)

        matches = regex.exec(result)
        fs.appendFileSync("logs.txt", `${text}\n${trial_txt}${result}\n\n`)

        tries_cnt--;
        await timer(CHATGPTTIME)
      }

      // no matches after all the trials.
      if (!matches) {
        const no_match_log = `no matches. Skipping the title '${title}'.\n`
        console.log(chalk.red(no_match_log))
        fs.appendFileSync("logs.txt", `${no_match_log}`)
        if (!skipped_titles.includes(title))
          fs.appendFileSync("skipped.csv", `${ids[index]},"${title_preprocessed}"\n`)
        continue;
      }

      if (matches) {
        const match_log = `✅ match was found for '${title}'.\nsaving output...\n`
        console.log(chalk.green(match_log))
        fs.appendFileSync("logs.txt", `${match_log}`)
      }

      const abstract = matches[1].trim().replace(/"/g, '""')
      const intro = matches[2].trim().replace(/"/g, '""')
      const conc = matches[3].trim().replace(/"/g, '""')

      fs.appendFileSync(OUTPUT, `"${title_preprocessed}","${abstract}","${intro}","${conc}"\n`)
    }
  }
  console.log("done.")
})()