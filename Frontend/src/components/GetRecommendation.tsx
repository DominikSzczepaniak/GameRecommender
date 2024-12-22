import React from 'react'
import { GameCard, GameCardProps } from './GameCard';
import { Select, SelectContent, SelectTrigger, SelectValue } from './ui/select';
import { getLanguageFile } from '@/helpers/language';
import { Button } from './ui/button';
import { Card, CardContent, CardFooter } from './ui/card';

interface GetRecommendationProps {
    recommendationEngines: { name: string, likeFunction: (recommendationNumber: number) => void, dislikeFunction: (recommendationNumber: number) => void, askForRecommendations: (userID: number) => GameCardProps[] }[];
    userID: number;
}

export const GetRecommendation = (props: GetRecommendationProps) => { //TODO: CSS
    const [visibleRecommendations, setVisibleRecommendations] = React.useState(false);
    const [chosenRecommendationEngine, setChosenRecommendationEngine] = React.useState(-5);
    const [recommendations, setRecommendations] = React.useState<GameCardProps[] | null>(null);
    const userID = props.userID;

    const translations = getLanguageFile();

    const askForRecommendations = () => {
        setRecommendations(props.recommendationEngines[chosenRecommendationEngine].askForRecommendations(userID));
        setVisibleRecommendations(true);
    }

    const recommendationLike = (recommendationNumber: number) => {
        props.recommendationEngines[chosenRecommendationEngine].likeFunction(recommendationNumber);
    }

    const recommendationDislike = (recommendationNumber: number) => {
        props.recommendationEngines[chosenRecommendationEngine].dislikeFunction(recommendationNumber);
    }

    return (
        <div className='flex justify-center items-center'>
            {(visibleRecommendations && recommendations !== null) ? (
                recommendations.map((recommendation, index) => {
                    return (<Card>
                        <CardContent>
                            <GameCard {...recommendation as GameCardProps}></GameCard>
                        </CardContent>
                        <CardFooter>
                            <button onClick={() => recommendationDislike(index)}>Dislike</button>
                            <button onClick={() => recommendationLike(index)}>Like</button>
                        </CardFooter>
                        </Card>)
                }))
             : (
                <>
                    <Select>
                        <SelectTrigger>
                            <SelectValue>{translations.getRecommendation.selectEngine}</SelectValue>
                        </SelectTrigger>
                        <SelectContent>
                            {props.recommendationEngines.map((engine, index) => (
                                <button onClick={() => { setChosenRecommendationEngine(index); }}>{engine.name}</button>
                            ))}
                        </SelectContent>
                    </Select>
                    {chosenRecommendationEngine === -5 ? <></> : <Button onClick={askForRecommendations}>{translations.getRecommendation.getGame}</Button>}
                </>
            )}
        </div>
    )
}
