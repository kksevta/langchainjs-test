const { ChatOpenAI } = require('@langchain/openai');
// const { Chroma } = require("@langchain/community/vectorstores/chroma")
const { OpenAIEmbeddings } = require("@langchain/openai");
const { ChatPromptTemplate } = require("@langchain/core/prompts");
const { BufferMemory } = require("langchain/memory");

const { ChromaClient } = require("chromadb");
const fs = require('fs');
const csv = require('csv-parser');
const OPEN_AI_API_KEY = ''
// Initialize OpenAI GPT-3.5 Turbo
const openai = new ChatOpenAI({
    openAIApiKey: OPEN_AI_API_KEY,
    modelName: 'gpt-3.5-turbo' // or 'gpt-4'
});

let chromaCollection;

async function initializeChroma() {
    const client = new ChromaClient({ path: "http://localhost:8000" });
    chromaCollection = await client.getOrCreateCollection({ name: "test-from-js" });
}


// Extract text from various file types
async function extractTextFromFile(filePath) {
    if (filePath.endsWith('.pdf')) {
        const dataBuffer = fs.readFileSync(filePath);
        const data = await pdfParse(dataBuffer);
        return data.text;
    } else if (filePath.endsWith('.csv')) {
        return parseCSV(filePath);
    } else if (filePath.endsWith('.txt')) {
        return fs.promises.readFile(filePath, 'utf8');
    } else {
        throw new Error('Unsupported file type');
    }
}

// Parse CSV files
async function parseCSV(filePath) {
    const results = [];
    return new Promise((resolve, reject) => {
        fs.createReadStream(filePath)
            .pipe(csv())
            .on('data', (data) => results.push(data))
            .on('end', () => resolve(results))
            .on('error', (error) => reject(error));
    });
}

// Get embeddings for text data
async function getEmbeddings(textArray) {
    const embeddings = new OpenAIEmbeddings({
        openAIApiKey: OPEN_AI_API_KEY, // Replace with your OpenAI API key
    });
    try {
        // Get embeddings for the array of texts
        const result = await embeddings.embedDocuments(textArray);
        console.log("Embeddings: ", result);
        return result;
    } catch (error) {
        console.error("Error fetching embeddings: ", error);
    }
    return embeddings;
}

// Store vectorized data in Chroma
async function storeDataInChroma(embeddings, data) {
    embeddings.forEach((embedding, index) => {
        chromaCollection.add({
            ids: [index.toString()], // ids should be an array
            embeddings: [embedding], // embeddings should be an array of arrays
            metadatas: [data[index]] // metadata should be an array of objects
        });
    });
}

// Main function to process a file
async function processFile(filePath) {
    const textData = await extractTextFromFile(filePath);
    const textArray = Array.isArray(textData) ? textData.map(row => JSON.stringify(row)) : [textData];
    const embeddings = await getEmbeddings(textArray);
    await storeDataInChroma(embeddings, textData);
}

// Handle a user query
async function handleQuery(query, queryEmbeddings) {
    // Initialize OpenAIEmbeddings
    const embeddings = new OpenAIEmbeddings({
        openAIApiKey: OPEN_AI_API_KEY,
    });

    // 1. Get the embeddings for the user's query
    const queryEmbedding = await embeddings.embedDocuments([query]);

    // 2. Search Chroma for relevant documents based on the query embedding
    const results = await chromaCollection.query({
        queryEmbeddings: queryEmbedding[0], // Use the first element of the array
        nResults: 5 // Number of relevant results to retrieve
    });

    // 3. Extract and format the context from Chroma results
    const metadatas = results.metadatas?.[0] || []; // Access the first array inside `metadatas`
    const context = metadatas.map(metadata => JSON.stringify(metadata)).join('\n'); 
    // 4. Prepare the prompt with the context
    const template = `You are an assistant that knows apartment details. Use the following context to answer the user's question. If the answer is not in the context, say "I don't know."
    Context:
    ${context}
    Human: ${query}
    AI:`;

    const prompt = ChatPromptTemplate.fromTemplate(template);

    // 5. Create the chain with the prompt and OpenAI model
    const chain = prompt.pipe(openai);

    // 6. Call the OpenAI API with the formatted prompt
    const response = await chain.invoke();

    return response;
}

// Example usage
async function run() {
    await initializeChroma();
    await processFile('./csv/my-data.csv'); // or .csv, .txt
    const answer = await handleQuery('Onepoint hosts which workshop');
    console.log(`Answer: ${answer?.content}`);
}

run();
