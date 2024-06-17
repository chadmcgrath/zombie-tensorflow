// this seems to fascilitate a little bit of exploring, rathert han just facing a wall waiting to die.
// seems to cause disco head
const rewardConfigs={
    default: {
        baseReward: 1,
        hitShotReward: 1,
        biteReward: -1,
        hitHumanReward: -1,
        missedShotReward: -.125,
        bumpWallReward: -.55,
        bumpScreenReward: -.75,
        bumpHumanReward: -.65,
        blockedVisionHuman: 0,
        blockedVisionWall: -.1,
        farVisionReward: 0,
        zombieProximityReward: 0
    },
    // some disco head her
    explore: {
        baseReward: 1,
        hitShotReward: 1,
        biteReward: -1,
        hitHumanReward: -1,
        missedShotReward: -.2,
        bumpWallReward: -.5,
        bumpScreenReward: -.5,
        bumpHumanReward: -.65,
        blockedVisionHuman: 0,
        blockedVisionWall: -.3,
        farVisionReward: 0,
        zombieProximityReward: -.2
    },
    attack: {
        baseReward: 1,
        hitShotReward: 1.5,
        biteReward: -1,
        hitHumanReward: -1,
        missedShotReward: -.2,
        bumpWallReward: -.55,
        bumpScreenReward: -.75,
        bumpHumanReward: -.65,
        blockedVisionHuman: -.5,
        blockedVisionWall: -.2,
        farVisionReward: .2,
        zombieProximityReward: -.25
    }
}
