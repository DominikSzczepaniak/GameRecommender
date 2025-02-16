import { getLanguageFile } from '@/helpers/language';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './ui/card';

export interface GameCardProps {
  imagePath: string;
  title: string;
  description: string;
  price: string;
  publisher: string;
  releaseDate: string;
}

export const GameCard = (props: GameCardProps) => {
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
          <img src={props.imagePath} alt={props.title} />
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
