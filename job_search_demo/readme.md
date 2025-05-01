# Job Search Demo

## Project Description

This project is a job search application that allows users to search for job titles using a query. The application uses a **FastAPI backend** to handle search requests and a **React frontend** to provide a user-friendly interface. The backend leverages **FAISS** and **Sentence Transformers** to perform similarity-based searches on job titles.

## Technologies Used

### Backend:
- **FastAPI**: For building the REST API.
- **FAISS**: For efficient similarity search.
- **Sentence Transformers**: For generating embeddings of job titles.
- **Python**: The programming language used for backend development.

### Frontend:
- **React**: For building the user interface.
- **Axios**: For making HTTP requests to the backend.
- **Vite**: For fast development and build processes.

## How It Works

1. **Backend**:
   - The backend loads job titles from a JSON file and generates embeddings using **Sentence Transformers**.
   - These embeddings are stored in a **FAISS index** for efficient similarity search.
   - When a user submits a query, the backend calculates the similarity between the query and the job titles, returning the most relevant results.

2. **Frontend**:
   - The frontend provides a search bar where users can input their queries.
   - The query is sent to the backend via an HTTP request.
   - The results are displayed as job cards on the frontend.

## Dependency Installation Steps

### Backend
1. Navigate to the backend directory:
   ```bash
   cd job_search_demo/backend
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Frontend
1. Navigate to the frontend directory:
   ```bash
   cd job_search_demo/frontend
   ```
2. Install the required dependencies:
   ```bash
   npm install
   ```

## Example Usage

1. Start the backend server:
   ```bash
   uvicorn app:app --reload
   ```

2. Start the frontend development server:
   ```bash
   npm run dev
   ```

3. Open the application in your browser and search for job titles using the search bar.

## Folder Structure

- **Backend**:
  - `app.py`: Main FastAPI application.
  - `embeddings/`: Contains modules for generating embeddings and managing the FAISS index.
  - `requirements.txt`: Lists the dependencies for the backend.

- **Frontend**:
  - `src/`: Contains React components and the main application logic.
  - `package.json`: Lists the dependencies for the frontend.

## Example Query and Output

**Query**: "Software Engineer"

**Output**: A list of job titles most similar to the query, displayed as cards on the frontend.

## Development Status

This project is currently under active development.
