import { GameData } from './Game';

export type RecommendationEngine = {
  name: string,
  askForRecommendations: () => Promise<GameData[]>,
  dislikeFunction: (appId: string) => void,
  likeFunction: (appId: string) => void,
};
