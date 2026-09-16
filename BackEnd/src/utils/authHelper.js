import jwt from "jsonwebtoken";

const getSignedToken = (id) => {
  const token = jwt.sign({ id }, process.env.JWT_SECRET, {
    expiresIn: process.env.JWT_EXPIRE,
  });
  return token;
};

export const createAndSendTokenCookie = (user, res) => {
  const token = getSignedToken(user._id);

  res.cookie("token", token, {
    expires: new Date(
      Date.now() + process.env.JWT_COOKIE_EXPIRES * 24 * 60 * 60 * 1000
    ),
    httpOnly: true,
    sameSite: process.env.NODE_ENV === "dev" ? "lax" : "none",
    secure: process.env.NODE_ENV !== "dev",
  });

  return token;
};
