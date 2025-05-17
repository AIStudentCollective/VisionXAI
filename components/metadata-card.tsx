interface Metadata {
  name: string;
  status: string;
  id: number | string;
  createdDate: string;
  institution: string;
  creator: string;
}

const MetadataCard: React.FC<Metadata> = ({
  name,
  status,
  id,
  createdDate,
  institution,
  creator,
}) => {
  return (
    <div className="bg-transparent text-white rounded-lg p-4 w-full max-w-sm shadow-md">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xl font-light">{name}</h2>
        <div className="px-3 py-0.5 rounded-full bg-indigo-500 text-white text-xs font-medium shadow hover:opacity-90 transition">
          {status}
        </div>
      </div>

      <div className="space-y-1.5 text-xs sm:text-sm font-light">
        <div className="flex justify-between">
          <span>Database ID:</span>
          <span className="font-normal">{id}</span>
        </div>
        <div className="flex justify-between">
          <span>Date Created:</span>
          <span className="font-normal">{createdDate}</span>
        </div>
        <div className="flex justify-between">
          <span>Institution:</span>
          <span className="font-normal">{institution}</span>
        </div>
        <div className="flex justify-between">
          <span>Creator:</span>
          <span className="font-normal">{creator}</span>
        </div>
      </div>
    </div>
  );
};

export default MetadataCard;