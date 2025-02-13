using Microsoft.EntityFrameworkCore;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace GameRecommender.Models;

public class User
{
    [Key]
    public Guid Id { get; }
    [MaxLength(64)]
    public string Username { get; set; }
    [MaxLength(64)]
    public string Email { get; set; }
    [MaxLength(32)]
    public string Password { get; private set; }

    public User(Guid anId, string aUsername, string anEmail, string aPassword)
    {
        Id = anId;
        Username = aUsername;
        Email = anEmail;
        Password = aPassword;
    }
}