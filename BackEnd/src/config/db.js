import mongoose from "mongoose";

const connectDB = async () => {
  const db = process.env.MONGO_URL.replaceAll(
    "<db_password>",
    process.env.DB_PASSWORD
  );

  try {
    const conn = await mongoose.connect(db, {
      serverSelectionTimeoutMS: 5000,
    });
    console.log(`MongoDB Connected: ${conn.connection.host}`);
  } catch (error) {
    console.log(error);
    process.exit(1);
  }
};

export default connectDB;
