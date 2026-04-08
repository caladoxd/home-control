FROM node:20-alpine

WORKDIR /app

COPY . .

RUN npm install --omit=dev

ENV NODE_ENV=production

EXPOSE 3000

CMD ["npm", "start"]