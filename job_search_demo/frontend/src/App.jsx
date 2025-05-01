import React, { useState } from 'react';
import axios from 'axios';
import SearchBar from './components/SearchBar';
import JobCard from './components/JobCard';

const App = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);

  const handleSearch = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/search/?query=${query}`);
      setResults(response.data.results || []); 
    } catch (error) {
      console.error("An error occurred during the search:", error);
      setResults([]);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Job Search</h1>
      <SearchBar query={query} setQuery={setQuery} handleSearch={handleSearch} />
      <div style={{ marginTop: '20px' }}>
        {results.length > 0 ? (
          results.map((job, index) => (
            <JobCard key={index} jobTitle={job} />
          ))
        ) : (
          <p>No search results found.</p> 
        )}
      </div>
    </div>
  );
};

export default App;
