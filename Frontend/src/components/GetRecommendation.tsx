import { getLanguageFile } from '@/helpers/language';
import React from 'react';
import { GameCard } from './GameCard';
import { Button } from './ui/button';
import { Card, CardContent, CardFooter } from './ui/card';
import { Select, SelectContent, SelectTrigger, SelectValue } from './ui/select';
import { GameData } from '@/models/Game';

interface GetRecommendationProps {
  recommendationEngines: {
    name: string,
    likeFunction: (appId: string) => void,
    dislikeFunction: (appId: string) => void,
    askForRecommendations: () => Promise<GameData[]>,
  }[];
}

export const GetRecommendation = (props: GetRecommendationProps) => { //TODO: CSS
  const [visibleRecommendations, setVisibleRecommendations] = React.useState(false);
  const [chosenRecommendationEngine, setChosenRecommendationEngine] = React.useState(-5);
  const [recommendations, setRecommendations] = React.useState<GameData[]>([]);

  const translations = getLanguageFile();

  const askForRecommendations = async () => {
    setRecommendations(await props.recommendationEngines[chosenRecommendationEngine].askForRecommendations());
    setVisibleRecommendations(true);
  };

  const recommendationLike = (appId: string) => {
    props.recommendationEngines[chosenRecommendationEngine].likeFunction(appId);
  };

  const recommendationDislike = (appId: string) => {
    props.recommendationEngines[chosenRecommendationEngine].dislikeFunction(appId);
  };

  return (
    <div className='flex justify-center items-center'>
      {(visibleRecommendations && recommendations !== null)
        ? (
          recommendations.slice(0,5).map((recommendation) => {
            return (
              <Card>
                <CardContent>
                  <GameCard {...recommendation}>
                  </GameCard>
                </CardContent>
                <CardFooter>
                  <button onClick={() => recommendationDislike(recommendation.appId)}>
                    Dislike
                  </button>
                  <button onClick={() => recommendationLike(recommendation.appId)}>
                    Like
                  </button>
                </CardFooter>
              </Card>
            );
          })
        )
        : (
          <>
            <Select>
              <SelectTrigger>
                <SelectValue>
                  {translations.getRecommendation.selectEngine}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {props.recommendationEngines.map((engine, index) => (
                  <button
                    onClick={() => {
                      setChosenRecommendationEngine(index);
                    }}>
                    {engine.name}
                  </button>
                ))}
              </SelectContent>
            </Select>
            {chosenRecommendationEngine === -5
              ? (
                <>
                </>
              )
              : (
                <Button onClick={askForRecommendations}>
                  {translations.getRecommendation.getGame}
                </Button>
              )}
          </>
        )}
    </div>
  );
};
