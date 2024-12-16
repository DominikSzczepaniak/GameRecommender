import React from 'react'
import { GameCard, GameCardProps } from './GameCard';
interface GetRecommendationProps {
    recommendationEngines: {name: string, likeFunction: () => void, dislikeFunction: () => void}[];
}

export const GetRecommendation = (props: GetRecommendationProps) => {
    const [visibleRecommendation, setVisibleRecommendation] = React.useState(0);
    const [recommendation, setRecommendation] = React.useState<GameCardProps>({title: "", description: "", imagePath: "", price: "", publisher: "", releaseDate: ""});
  return (
    <div>
        {visibleRecommendation ? (
            <div>
                <button onClick={props.recommendationEngines[visibleRecommendation].likeFunction}>Like</button>
                <GameCard props={recommendation}></GameCard>
            </div>
        ) : (<></>)}

    </div>
  )
}
