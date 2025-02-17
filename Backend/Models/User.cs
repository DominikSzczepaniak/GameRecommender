using Microsoft.EntityFrameworkCore;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace GameRecommender.Models;

public class User
{
    public Guid Id { get; set; }
    public string Username { get; set; }
    public string Email { get; set; }
    public string Password { get; set; }
    public User () {}

    public User(Guid anId, string aUsername, string anEmail, string aPassword)
    {
        Id = anId;
        Username = aUsername;
        Email = anEmail;
        Password = aPassword;
    }
}