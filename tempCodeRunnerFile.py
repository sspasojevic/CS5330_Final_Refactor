Forward
        if self.wnd.is_key_pressed(keys.L): object.position[2] += speed # Back

        if self.wnd.is_key_pressed(keys.K): object.position[0] -= speed # Left
        if self.wnd.is_key_pressed(keys.SEMICOLON): object.position[0] += speed # Right
