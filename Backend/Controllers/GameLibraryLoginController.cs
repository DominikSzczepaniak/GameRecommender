using GameRecommender.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace GameRecommender.Controllers;
[ApiController]
[Route("gameLibraryLogin")]
public class GameLibraryLoginController(IGameLibrary gameLibrary) : Controller
{
    [HttpGet]
    public async Task<IActionResult> GetSteamProfile(Guid userId, string steamLink)
    {
        try
        {
            await gameLibrary.SetUserSteamProfile(userId, steamLink);
            return Ok();
        }
        catch (ArgumentException ex)
        {
            return BadRequest(ex.Message);
        }

    }
}