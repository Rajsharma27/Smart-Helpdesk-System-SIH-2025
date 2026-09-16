import { promisify } from "util";
import jwt from "jsonwebtoken";

import User from "../models/User.js";

const authMiddleware = {
  protect: async (req, res, next) => {
    try {
      // Take out authorization header and cookies recieved
      const { authorization } = req.headers;

      let token;
      // Check if have authorization header
      if (!authorization) {
        res.status(401).json({ msg: "Unauthorized (No token)" });
      }
      // Check if have token
      if (authorization && authorization.startsWith("Bearer")) {
        token = authorization.split(" ")[1];
      }

      if (!token) {
        res.status(401).json({ msg: "Unauthorized (No token)" });
      }

      const decodedUser = await promisify(jwt.verify)(
        token,
        process.env.JWT_SECRET
      );

      const user = await User.findById(decodedUser.id);

      if (!user) {
        res.status(401).json({ msg: "Unauthorized (User not found)" });
      }

      req.user = user;

      next();
    } catch (error) {
      console.error(error.message);
      res.status(500).json({ msg: "Server Error" });
    }
  },
};

export default authMiddleware;
