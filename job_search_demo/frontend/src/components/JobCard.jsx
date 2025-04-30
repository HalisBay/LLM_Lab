import React from 'react';

const JobCard = ({ jobTitle }) => {
  return (
    <div style={{ border: '1px solid #ccc', margin: '10px', padding: '10px' }}>
      {jobTitle}
    </div>
  );
};

export default JobCard;
