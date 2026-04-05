FROM node:20-alpine

WORKDIR /app

COPY package*.json ./

RUN npm install --legacy-peer-deps

COPY . .

RUN npm run build --if-present

ENV NODE_ENV=production

EXPOSE 3000

CMD ["npm", "start"]
