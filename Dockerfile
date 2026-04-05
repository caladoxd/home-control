FROM node:20-alpine

WORKDIR /app

COPY . .

RUN npm install --legacy-peer-deps

RUN npm run build --if-present

ENV NODE_ENV=production

EXPOSE 3000

CMD ["npm", "start"]
