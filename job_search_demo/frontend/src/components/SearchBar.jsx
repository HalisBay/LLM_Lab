import React from 'react';

const SearchBar = ({ query, setQuery, handleSearch }) => {
  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div>
      <input 
        type="text" 
        value={query} 
        onChange={(e) => setQuery(e.target.value)} 
        onKeyPress={handleKeyPress} // Enter tuşuna basıldığında arama yap
        placeholder="İş unvanı ara..." 
        style={{ marginRight: '10px', padding: '8px' }}
      />
      <button onClick={handleSearch} style={{ padding: '8px 15px' }}>search</button>
    </div>
  );
};

export default SearchBar;
