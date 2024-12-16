import React from 'react'
import { GameCard, GameCardProps } from './GameCard';
import { Select, SelectContent, SelectTrigger, SelectValue } from './ui/select';
import { getLanguageFile } from '@/helpers/language';
import { Button } from './ui/button';

interface GetRecommendationProps {
    recommendationEngines: { name: string, likeFunction: () => void, dislikeFunction: () => void, askForRecommendation: (userID: number) => GameCardProps }[];
    userID: number;
}

export const GetRecommendation = (props: GetRecommendationProps) => { //TODO: CSS
    const [visibleRecommendation, setVisibleRecommendation] = React.useState(false);
    const [chosenRecommendationEngine, setChosenRecommendationEngine] = React.useState(-5);
    const [recommendation, setRecommendation] = React.useState<GameCardProps | null>(null);
    const userID = props.userID;

    const translations = getLanguageFile();

    const askForRecommendation = () => {
        setRecommendation(props.recommendationEngines[chosenRecommendationEngine].askForRecommendation(userID));
        setVisibleRecommendation(true);
    }

    const resetRecommendation = () => {
        setVisibleRecommendation(false);
        setRecommendation(null);
    }

    const recommendationLike = () => {
        props.recommendationEngines[chosenRecommendationEngine].likeFunction();
        resetRecommendation();
    }

    const recommendationDislike = () => {
        props.recommendationEngines[chosenRecommendationEngine].dislikeFunction();
        resetRecommendation();
    }

    return (
        <div>
            {(visibleRecommendation && recommendation !== null) ? (
                <div>
                    <button onClick={recommendationDislike}>Dislike</button>
                    <GameCard {...recommendation as GameCardProps}></GameCard>
                    <button onClick={recommendationLike}>Like</button>
                </div>
            ) : (
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
                    {chosenRecommendationEngine === -5 ? <></> : <Button onClick={askForRecommendation}>{translations.getRecommendation.getGame}</Button>}
                </>
            )}
        </div>
    )
}
