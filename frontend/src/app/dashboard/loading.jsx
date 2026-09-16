import { BounceLoader } from "react-spinners";

const loading = () => {
  return (
    <div className="h-screen w-screen flex items-center justify-center">
      <BounceLoader color="#4A90E2" />
    </div>
  );
};

export default loading;
