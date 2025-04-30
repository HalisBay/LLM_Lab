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
      // Eğer backend { results: [...] } gibi bir nesne döndürüyorsa:
      // setResults(response.data.results || []);
    } catch (error) {
      console.error("Arama sırasında hata oluştu:", error);
      setResults([]); // Hata durumunda sonuçları temizle
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>İş Arama</h1>
      <SearchBar query={query} setQuery={setQuery} handleSearch={handleSearch} />
      <div style={{ marginTop: '20px' }}>
        {results.length > 0 ? (
          results.map((job, index) => (
            <JobCard key={index} jobTitle={job} />
          ))
        ) : (
          <p>Arama sonucu bulunamadı.</p> 
        )}
      </div>
    </div>
  );
};

export default App;
