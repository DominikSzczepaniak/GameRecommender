import { getGamePhoto } from '@/helpers/gamePhotos';
import { getLanguageFile } from '@/helpers/language';
import { GameData } from '@/models/Game';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './ui/card';

export const GameCard = (props: GameData) => {
  const translations = getLanguageFile();
  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>
            {translations.gameCard.title}
          </CardTitle>
          <CardDescription>
            {props.title}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {getGamePhoto(props.appId)}
          <p>
            {translations.gameCard.description}: {props.description}
          </p>
          <p>
            {translations.gameCard.price}: {props.price}
          </p>
        </CardContent>
        <CardFooter>
          <p>
            {translations.gameCard.publisher}: {props.publisher}
          </p>
          <p>
            {translations.gameCard.releaseDate}: {props.releaseDate}
          </p>
        </CardFooter>
      </Card>
    </>
  );
};
